#!/usr/bin/env python3
"""
Setup script for the Finance Analyst AI Agent.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="finance_analyst_agent",
    version="1.0.0",
    author="Likith Pranai",
    author_email="your.email@example.com",
    description="A professional-grade AI agent for comprehensive financial analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Finance_Analyst_AI_Agent",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "yfinance",
        "pandas",
        "numpy",
        "matplotlib",
        "plotly",
        "dash",
        "streamlit",
        "google-generativeai",
        "python-dotenv",
        "requests",
        "scikit-learn",
        "prophet",
        "redis",
    ],
    entry_points={
        "console_scripts": [
            "finance-agent=finance_agent:main",
        ],
    },
)
