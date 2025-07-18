# Finance Analyst AI Web UI

A modern, GROK AI-style web interface for the Finance Analyst AI Agent. This web UI provides an intuitive chat interface for interacting with the Finance Analyst AI Agent's powerful financial analysis capabilities.

## Features

- **Modern Chat Interface**: Clean, responsive design inspired by GROK AI
- **Real-time Communication**: WebSocket integration for streaming responses
- **Dark/Light Mode**: Toggle between dark and light themes
- **Markdown Support**: Rich text formatting with code syntax highlighting
- **Interactive Elements**: Example queries and conversation history
- **Responsive Design**: Works on desktop and mobile devices

## Architecture

The web UI consists of two main components:

1. **Frontend**: React application with TypeScript and modern styling
   - React with TypeScript for type safety
   - Framer Motion for smooth animations
   - React Markdown for rendering formatted responses
   - WebSocket integration for real-time communication

2. **Backend**: FastAPI server that connects to the Finance Analyst AI Agent
   - FastAPI for high-performance API endpoints
   - WebSocket support for streaming responses
   - Integration with the existing Finance Analyst AI Agent

## Getting Started

### Prerequisites

- Node.js 16+ for the frontend
- Python 3.8+ for the backend
- Finance Analyst AI Agent dependencies

### Installation

1. **Install backend dependencies**:
   ```bash
   cd web_ui/backend
   pip install -r requirements.txt
   ```

2. **Install frontend dependencies**:
   ```bash
   cd web_ui/frontend
   npm install
   ```

### Running the Application

#### Development Mode

1. **Start the backend server**:
   ```bash
   cd web_ui/backend
   uvicorn main:app --reload
   ```

2. **Start the frontend development server**:
   ```bash
   cd web_ui/frontend
   npm start
   ```

3. Open your browser and navigate to `http://localhost:3000`

#### Using Docker

For a containerized setup, use Docker Compose:

```bash
cd web_ui
docker-compose up
```

This will start both the frontend and backend services.

## Usage

1. **Start a conversation**: Type your financial query in the input box at the bottom of the screen
2. **Example queries**: Click on any of the example queries on the welcome screen
3. **View conversation history**: Access previous conversations from the sidebar
4. **Toggle dark/light mode**: Use the theme toggle button in the sidebar

## Integration with Finance Analyst AI Agent

The web UI seamlessly integrates with the existing Finance Analyst AI Agent, providing access to all its capabilities:

- Technical indicators (RSI, MACD, OBV, A/D Line, ADX)
- Fundamental analysis with financial ratios and statements
- Portfolio management and optimization
- Predictive analytics and backtesting
- Real-time data for stocks, cryptocurrencies, and forex markets

## Customization

- **Theme Colors**: Modify the CSS variables in `src/styles/index.css`
- **Layout**: Adjust the component styling in their respective CSS files
- **API Endpoints**: Configure the backend connection in the ChatInterface component

## License

This project is part of the Finance Analyst AI Agent and follows the same licensing terms.

---

Built with ❤️ for financial analysis professionals and enthusiasts
