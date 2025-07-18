# Contributing to Finance Analyst AI Agent

Thank you for your interest in contributing to the Finance Analyst AI Agent project! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to foster an open and welcoming environment.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with the following information:

1. A clear, descriptive title
2. A detailed description of the bug
3. Steps to reproduce the bug
4. Expected behavior
5. Actual behavior
6. Screenshots (if applicable)
7. Environment information (OS, Python version, etc.)

### Suggesting Features

We welcome feature suggestions! Please create an issue with:

1. A clear, descriptive title
2. A detailed description of the feature
3. The problem the feature would solve
4. Any alternative solutions you've considered
5. Any additional context or screenshots

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Add tests for your changes
5. Run the tests (`python -m tests.run_tests`)
6. Commit your changes (`git commit -m 'Add some feature'`)
7. Push to the branch (`git push origin feature/your-feature-name`)
8. Create a new Pull Request

## Development Setup

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/Finance_Analyst_AI_Agent.git
   cd Finance_Analyst_AI_Agent
   ```

2. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. Run tests
   ```bash
   python -m tests.run_tests
   ```

## Project Structure

- `finance_analyst_agent.py`: Main agent implementation
- `finance_agent.py`: Unified entry point
- `tools/`: Tool modules for different functionalities
- `tests/`: Test suite
- `models/`: Machine learning models
- `data/`: Data storage
- `utils/`: Utility functions
- `config/`: Configuration files
- `docs/`: Documentation

## Coding Standards

- Follow PEP 8 style guide
- Write docstrings for all functions, classes, and modules
- Add type hints where appropriate
- Write unit tests for new functionality
- Keep functions small and focused on a single task

## Testing

- All new features should include tests
- Run the test suite before submitting a pull request
- Aim for high test coverage

## Documentation

- Update documentation for any changes to the API
- Document new features
- Keep the README up to date

Thank you for contributing to the Finance Analyst AI Agent project!
