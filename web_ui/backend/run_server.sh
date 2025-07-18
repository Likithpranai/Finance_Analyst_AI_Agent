if [ -d "../../venv" ]; then
  echo "Activating virtual environment..."
  source ../../venv/bin/activate
fi

if [ -z "$GEMINI_API_KEY" ]; then
  if [ -f "../../.env" ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' ../../.env | xargs)
  else
    echo "Warning: GEMINI_API_KEY not set and no .env file found."
    echo "The Finance Analyst AI Agent requires a Gemini API key to function."
  fi
fi

echo "Starting FastAPI server..."
uvicorn main:app --reload --host 0.0.0.0 --port 8000
